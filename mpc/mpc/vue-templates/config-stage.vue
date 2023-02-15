<template>
    <div>
        <v-row no-gutters>
            <v-icon @click="update_cdict()">mdi-update</v-icon>
            <v-chip @click="show_cdict=!show_cdict">show {{stageid}} dict</v-chip>
        </v-row>
        <v-row>
            <v-col>
                <stage-config></stage-config>
            </v-col>
            <v-col v-if="show_cdict"><kbd>{{cdict}}</kbd></v-col>
        </v-row>
    </div>
</template>