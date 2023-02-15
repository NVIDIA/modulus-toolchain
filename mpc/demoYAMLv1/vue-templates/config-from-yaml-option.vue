<template>
    <div class="ma-0">
        <v-row v-if="['option', 'group'].includes(schema[key]['type'])" class="ma-0">
            <v-col class="ma-0 pa-0">
                <v-row>
                    <v-divider></v-divider>
                </v-row>
                <v-row class="ma-0 pa-0" align="center">

                    <v-col cols="auto">
                        <v-icon @click="visible=!visible">{{visible ? 'mdi-eye' : 'mdi-eye-off'}}</v-icon>
                    </v-col>
                    <!-- <v-col cols="auto">
                        <v-icon @click="update_cdict()">mdi-update</v-icon>
                    </v-col> -->
                    <v-col>
                        <h3>{{schema[key].label || key}}</h3>
                    </v-col>
                    <v-col cols="auto" v-if="schema[key].type==='option'">
                        <v-chip small>{{cdict[key].__selected__ || ''}}</v-chip>
                    </v-col>
                </v-row>
                <v-row v-if="visible" class="ma-0">
                    <v-card class="ma-1">
                        <v-row v-if="schema[key].type==='option'" no-gutters>
                            <v-chip @click="update_choice(optk)" small outlined class="ma-1"
                                v-for="(item, optk) in schema[key].choices"
                                :color="optk==cdict[key].__selected__ ? 'green': ''">{{optk}}</v-chip>
                        </v-row>
                        <v-row no-gutters>
                            <yaml-items></yaml-items>
                        </v-row>
                    </v-card>
                </v-row>
            </v-col>
        </v-row>
        <v-row v-if="!['option', 'group'].includes(schema[key]['type'])" class="ma-1">
            <v-btn small outlined v-if="schema[key].type==='bool'" :color="cdict[key] ? 'green':'red'"
                @click="cdict[key]=!cdict[key]">{{key}}:<v-icon dark right>
                    {{cdict[key] ? "mdi-checkbox-marked-circle" : "mdi-cancel"}}
                </v-icon>
            </v-btn>

            <v-textarea outlined rows=2 v-if="schema[key].type==='textarea'" :label="schema[key].label || key"
                v-model="cdict[key]">
            </v-textarea>
            <v-text-field number v-if="!['bool', 'fixed', 'textarea'].includes(schema[key]['type'])"
                :hint="schema[key].hint ||schema[key].type" :label="schema[key].label || key" v-model="cdict[key]">
            </v-text-field>
            <div><kbd v-if="schema[key].type==='fixed'">{{key}}: {{cdict[key]}}</kbd></div>
        </v-row>

    </div>
</template>