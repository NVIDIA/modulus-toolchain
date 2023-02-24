<template>
    <div>
        <v-card>
            <v-card-title>Plots</v-card-title>
            <v-divider></v-divider>
            <v-row>
                <v-col>
                    <v-row no-gutters>
                        <v-col class="pl-2">

                            <v-chip class="mr-1" v-for="(val, key) in stages" :key="key"
                                :color="val.plot ? 'green':'red'" @click="switch_stage(key)" v-if="val.trained">
                                {{key}}
                            </v-chip>

                        </v-col>
                    </v-row>
                    <v-row>
                        <v-col>
                            <plt-figure></plt-figure>
                        </v-col>
                    </v-row>
                </v-col>
            </v-row>
        </v-card>

    </div>
</template>